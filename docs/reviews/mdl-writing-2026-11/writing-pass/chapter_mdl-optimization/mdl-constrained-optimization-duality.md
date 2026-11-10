# Writing pass: Constrained Optimization

File: `chapter_mdl-optimization/mdl-constrained-optimization-duality.md`.
Prose only; no cell, cell ID, figure include or slide changed.

## Words

Prose words (checker count, body and exercises): 7,612 before, 7,561 after
(-51). The cuts (retold Lagrange proof, two price paragraphs, scaling remark,
prox existence detour, coordinate-descent prose, recap) removed about 600
words; the additions the assignment asked for (full Slater definition, the
account of the 974-step run, the $\ell_1$ scope statement, adversarial
examples, the exact softmax statement, a ranking sentence) put about 550
back. The four coordinate-descent equations carry labels that may not be
removed, which limits how far that subsection can shrink in this pass.

## Correctness changes

- **CON-6 (Slater).** Said: strictly feasible means $g_i(\bar{\mathbf x}) < 0$
  for all $i$. Now: displayed definition `eq_mdl-opt-slater`, strict for the
  non-affine $g_i$, $\le 0$ for the affine ones, $h_j = 0$; a parenthesis adds
  "relative interior of the domain" for functions not defined on all of
  $\mathbb R^n$ (DUA-7); one sentence that all-affine problems need only a
  feasible point; the necessity statement now names its reason (strong
  duality, `subsec_mdl-strong-duality`). Verified against Boyd and
  Vandenberghe 5.2.3 and 5.5.3 from memory, and against the duality
  section's current proposition, which already reads "relative interior" and
  "not affine".
- **CON-8 (974 steps).** Said: "a few steps contract far more. The run reaches
  distance $10^{-6}$ after 974 steps ... guarantees this only after about
  7,660". Now: the guarantee 7,660 is stated before the cell as what $\rho$
  gives; after the cell the text says the count is not a rate, describes the
  bouncing, and gives the cause (curvature $500.5$ along the line, $\eta$ its
  inverse, exact Newton step the first time two consecutive steps are
  projected). Verified by rerunning the cell's loop with a trace: projected
  flags alternate True/False from step 1, the first two consecutive projected
  steps are 973 and 974, last ratio $6\times10^{-14}$, ten steps contract by
  more than 10%, $\mathbf v^\top\mathbf A\mathbf v = 500.5 = 1/\eta$; guarantee
  recomputed as 7,656. With $\eta = 1.9/1001$ or $1/1000$ the run does not
  finish in one step after the first consecutive projection (3 and 10 more
  steps).
- **CON-10 (softmax).** Said: "Softmax is the same projection with an entropy
  term in place of the squared distance." Now: sparsemax maximizes
  $\mathbf y^\top\mathbf x - \tfrac12\|\mathbf x\|^2$ over $\Delta$, softmax
  maximizes $\mathbf y^\top\mathbf x + H(\mathbf x)$, with
  `eq_mdl-opt-lse-entropy`. Verified by expanding the square.
- **CON-11 and the assignment's $\ell_1$ item.** Said: $\lambda$ "plays the
  role of the multiplier of a constraint $\|\mathbf w\|_1 \le c$", with no
  proof anywhere. Now: the direction penalty to constraint is proved in one
  line (for feasible $\mathbf w$,
  $s(\mathbf w) = F(\mathbf w) - \lambda\|\mathbf w\|_1 \ge F(\mathbf w_\lambda) - \lambda c = s(\mathbf w_\lambda)$;
  needs no convexity), and the text says the converse is not proved and that
  `subsec_mdl-weight-decay-duality` treats the squared $\ell_2$ norm only.
  Verified by the algebra and numerically (three seeds, 20,000 feasible
  points each, none below $s(\mathbf w_\lambda)$). **Not proved in the
  appendix:** for every budget $c$ there is a $\lambda$ (needs Slater and KKT
  with subgradients). If the duality writer adds the general convex-penalty
  sentence (DUA-11), the last clause here should cite it instead.
- **CON-13.** The prox existence argument (lower semicontinuity, Weierstrass,
  unexplained quadratic growth) replaced by the definition of proper closed
  convex with "closed epigraph" and "1-strongly convex, exactly one
  minimizer", cited to Parikh and Boyd.
- **CON-14.** "within about $0.1$" is now "within about $0.05$" (stored
  output 0.040; my three reruns 0.035 to 0.047).
- **CON-5.** The KKT conditions are now introduced as holding "at a minimum
  where a constraint qualification holds".
- **CON-15.** I first wrote the finished comparison ($L/\bar L$), then removed
  it: for this lasso $\mu/\bar L \approx 0.55$ is not small, the rate is for a
  smooth objective and random order, and "exact minimization" is not a
  difference (the $1/L_j$ prox step is already exact for a quadratic smooth
  part). The text now reports the measured factor and says no stated result
  predicts it. The old explanation ("the exact one-dimensional minimizer
  needs no step size", "uses the residual already updated") is gone.
- **CON-16.** "embedding tables" dropped.

No finding disagreed with.

## Before and after (verbatim)

1. Before: "This section asks how to recognize the best point when the
   variables must stay in a set, and how to compute it." After: "How do we
   recognize the best point when the variables must stay in a set, and how do
   we compute it?"
2. Before: "We test the proposition on the chapter's running example, the
   quadratic ..." After: "How tight is the factor $\rho$? Take the quadratic
   ..."
3. Before: "The cell runs PGD from $(3, 4)$, using the half-space projection
   above, and asks whether the distance to $\mathbf{x}^\star$ ever shrinks by
   less than $\rho$ in a step, and whether the run ends at the KKT point."
   After: "The run below records the worst contraction it sees in a step, the
   number of steps it needs, and the gradient where it stops."
4. Before: "The cell below projects six random scores. It then checks the KKT
   residuals for the computed $\tau$ and for a wrong one ..." After: "A
   computed projection should be tested, and the KKT residuals are the obvious
   candidates. The cell evaluates them for six random scores, at the computed
   $\tau$ and at a wrong one ..."
5. Before: "On the same lasso, does one prox per step find the planted zeros?"
   After: "On the lasso the prox is soft-thresholding with threshold
   $\eta\lambda$, so every coordinate that the gradient step leaves within
   $\eta\lambda$ of zero is set to exactly zero. That is the size at which the
   null coordinates of the subgradient run kept changing sign, so ISTA should
   hold them at zero."
6. Before: "The function $\varphi$ is $f$ plus a penalty that is nonpositive
   on the feasible set and zero at $\mathbf{x}^\star$." After: "The proof
   contains an idea larger than the proposition. For *any* fixed multipliers
   with $\boldsymbol{\lambda} \ge 0$, the first inequality of the chain holds,
   so the unconstrained minimum of the Lagrangian is a lower bound on $f$ over
   the feasible set; at the KKT multipliers the bound is attained." (CON-7)

"Does it find the five zeros?" is kept as it was.

## Repetition

Removed or reduced to a reference:
- Lagrange proof paragraph (CON-2): now cites `subsec_mdl-lagrange`; the
  sign reconciliation is one sentence. The proposition and its label stay.
- Multiplier as a price (CON-3): the hyperplane paragraph is one sentence
  with the number $0.02$ and a reference to `subsec_mdl-shadow-prices`; the
  $p^\star(s)$ computation after the clipping cell is deleted (owner:
  Duality).
- "A multiplier depends on how its constraint is written" (CON-4): deleted
  here (owner: Duality).
- Subgradient definition and the test $0 \in \partial f$ (CON-12): one clause
  with `sec_mdl-convexity` (the "Subgradients" subsection there has no label).
- Restatement of the KKT sufficiency proof in words; restatement of the PGD
  proof in words; the "We can now" recap (CON-18). Its last three sentences
  stay as the ending. The Summary gained one sentence (ball, half-space,
  simplex reduce to one scalar) from the recap.

Kept because this file owns them: the definition of Slater's condition; the
tangency figure; KKT sufficiency; projection, PGD, simplex, prox.

## Ranking, propositions, labels

- No Proposition block demoted: the four (Lagrange condition, KKT
  sufficiency, nonexpansiveness, PGD contraction) are all cited later.
- Added: an opening sentence naming the three results that carry the section;
  "the result of this section that we will use most" before KKT sufficiency;
  a first-reading skip notice on coordinate descent.
- **Labels added:** `eq_mdl-opt-slater` (the displayed definition) and
  `subsec_mdl-constraint-qualifications` (the subsection that holds it).

## Split markers

None; the file is not to be split.

## Cells to cut or redesign on the development machine

- `mdl-constrained-optimization-duality-soft-thresholding-and-the-proximal-operator-3`
  (no finding ID; brief section 4): brute-force check of the proved
  soft-threshold formula. Prose reduced to two sentences. Cut.
- `mdl-constrained-optimization-duality-projected-gradient-descent` (CON-8):
  choose a start on the boundary or $\eta = 1/1000$ so that the rate can be
  read against $\rho$; the present run terminates by coincidence.
- `mdl-constrained-optimization-duality-active-and-inactive-constraints` (no
  ID): evaluates the closed form just derived; harmless, short, could go.
- `mdl-convexity-coordinate-descent` (CON-15, CON structural 1): cut with the
  subsection if coordinate descent is reduced to a paragraph.
- `constrained-imports` (CON-19): unused framework imports.

## Other deferred

- Coordinate descent to a single paragraph (CON structural 1, CON-17 in
  full): needs the three general equations and their labels removed;
  `eq_mdl-quadratic-coordinate` is used by exercise 10. I reordered the prose
  so the lasso comes first (CON-17) and cut it from about 620 to about 330
  words.
- Exercise 8 (Moreau) to the Duality section; exercise 10(c) prints its own
  answers; stale exercise label on exercise 2: no moves between files, labels
  may not be renamed.
- Retitling the fourth part ("Penalties with Kinks"): not done, the heading
  has no label and slides refer to its content; author's call.
- **Notation, logged and not changed:** the clipped point is $\mathbf y$ here
  and $\mathbf x_0$ in the Duality section (DUA-1). $\mu$ is the Lagrange
  multiplier in `subsec_mdl-lagrange` and the strong-convexity constant here
  (CON-2); the calculus chapter still uses the minus-sign convention.
  $\lambda$ is both the inequality multiplier and the lasso weight (intended,
  now with its scope stated).
- The duality section currently cites Slater's condition by
  `subsec_mdl-kkt-conditions` (the `##` part); it resolves, but
  `eq_mdl-opt-slater` or `subsec_mdl-constraint-qualifications` is the exact
  target.

## Claims added or changed

1. Slater's condition in refined form, with the relative-interior clause and
   the all-affine remark.
2. KKT necessity under Slater "because of strong duality".
3. For any $\boldsymbol\lambda \ge 0$ the unconstrained minimum of the
   Lagrangian is a lower bound on $f$ over the feasible set.
4. The account of the PGD run (bouncing, curvature 500.5, exact Newton step,
   974 is a coincidence). The guarantee "about 7,660" is kept (recomputed:
   7,656); it is not printed by a cell.
5. Adversarial examples as projected gradient ascent under an $\ell_\infty$
   budget, projection = coordinatewise clip (Madry et al.).
6. Sparsemax and softmax as maximizers over the simplex.
7. Penalty to constraint for the lasso; the converse declared unproved.
8. Prox objective is 1-strongly convex with a unique minimizer (by citation).
9. "within about 0.05"; "Nothing proved in this section predicts that
   factor"; known coordinate-descent rates are for random order (Nesterov
   2012, as already cited).
10. $0.02 = -\nu^\star \cdot 0.1$ (exact increase 0.0202).

Unprinted numbers kept: 7,660 (recomputed), 4.47 (recomputed), 500.5
(recomputed), "$L$ between about 130 and 190" and "four to six sweeps, ten to
twenty-five steps" (ranges from the original text; stored draw 161.5, 4 and
22; $L$ over my 200 draws consistent).

## Slides against the body

- Slides still teach "read a multiplier as the price of its constraint" and
  "A multiplier is a price: $\partial p^\star/\partial b = -\nu^\star$"; the
  body now gives the number and defers the statement to the Duality section.
- Slide: "Slater (one strictly feasible point)"; the body now has the refined
  form.
- The summary slide says ISTA and coordinate descent give exact lasso zeros;
  still true. No slide quotes 974, 7,660 or "within 0.1".
