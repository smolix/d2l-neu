# Writing pass log: Duality

Source: `chapter_mdl-optimization/mdl-lagrangian-duality.md`. Not split.
Lightest hand of the chapter, as assigned: correctness items, one ending,
cell introductions, labels in place of "the previous section".

## Words

Prose words (checker count, body and exercises): 8562 before, 8556 after.
Checker: "OK: only prose changed", no em-dash warning; `lint_source.py` clean.

## Correctness changes

- **DUA-7 (i), Slater defined twice.** Before: the proposition spelled out its
  own definition (strict for non-affine $g_i$, $\le$ for affine ones,
  $h_j = 0$, point "in the interior of $\mathcal{D}$"). Now: one sentence,
  "convex ... and satisfies Slater's condition of
  :numref:`subsec_mdl-kkt-conditions`: some feasible point in the relative
  interior of $\mathcal{D}$ has $g_i < 0$ for every $g_i$ that is not
  affine." "Interior" became "relative interior" (the form in Boyd and
  Vandenberghe 5.2.3; with $\mathcal{D} = \mathbb{R}^n$ nothing changes).
  The reference is to the existing subsection label, since the definition
  itself has no label visible to me.
- **DUA-7 (ii).** "the constraint qualification announced in the previous
  section" is now "the constraint qualification that
  :numref:`subsec_mdl-kkt-conditions` introduced for convex problems". The
  `:citet:` of Slater (1950) went with the old sentence; the citation lives
  with the definition in the constrained section.
- **DUA-7 (iii), SVM.** Before: "Slater's condition holds for strictly
  separable data: scale ... until all margins exceed $1$." Now: "The
  constraints are affine, so Slater's condition needs only a feasible point,
  and scaling any separating $(\mathbf{w}, b)$ up until every margin is at
  least $1$ gives one." Exercise 3 ("Slater's condition in its refined form")
  now reads "needs only a feasible point".
- **DUA-5 / DUA-17, closest-point numbers.** Before: the half-plane solution
  "is the projection ... of the previous section's closest-point example"
  (that example has $\mathbf{a} = (3,4)$, $b = 5$; this one $(1,2)$, $-1$).
  Now: "the projection $(b/\|\mathbf{a}\|^2)\mathbf{a}$ of
  :numref:`subsec_mdl-closest-point`, here for a different line". Exercise
  6(b) says "here with $\mathbf{a} = (1, 2)$, $b = 1$". I did not change the
  numbers to $(3,4)$, $5$ as the review suggests: a slide carries
  $\mathbf{a} = (1,2)$, $b = -1$ and $d^\star = \tfrac1{10}$, and slides are
  frozen. Deferred (see below).
- **DUA-9, one curve.** New sentence before the shadow-price proposition: the
  lower edge of $\mathcal{A}$ is the graph of $p^\star(u)$ and the supporting
  line of slope $-\lambda^\star$ lies below it. Verified: the lowest $t$ in
  $\mathcal{A}$ at level $u$ is $\inf\{f : g \le u\} = p^\star(u)$. The
  clipping paragraph no longer rederives the curve; it cites
  $p^\star(u) = \tfrac12(5 - \sqrt{1 + 2u})^2$ from after the Slater proof
  (same function as the old $p^\star(s)$ with $s = \tfrac12 + u$; slope
  $-(5 - r)/r = -4$ at $r = 1$; value $6.43$ at $u = \tfrac12$ against the
  bound $6$, recomputed). $\rho$ renamed $r$ after the proof.
- **DUA-11, $\ell_1$.** New paragraph for a general convex penalty $\Omega$.
  Checked: penalty to constraint is the two-line inequality in the text
  (needs no convexity); constraint to penalty uses Slater (budget above
  $\min\Omega$) and the saddle-point proposition of this section, for convex
  $\ell$ and an attained constrained minimum.
- No finding disagreed with. DUA-1 not applied by instruction (below).

## Six before/after pairs

1. Before: "Do these values, and the best bound of $8$, follow from the
   definition $f + \lambda g$ without the closed form for $q$? The cell below
   minimizes the Lagrangian through $\mathbf{x}(\lambda)$, evaluates the bound
   on a grid of $\lambda$, and plots it against $p^\star$."
   After: "The cell evaluates $f + \lambda g$ at $\mathbf{x}(\lambda)$ on a
   grid of $\lambda$ and plots the bounds against $p^\star$. The numbers will
   be the ones above; the shape of the curve is what to look at."
2. Before: "Every value is a bound, the best bound is the optimum, and the
   $\lambda$ that attains it is the KKT multiplier."
   After: "Concavity is not a property of the disk: the next subsection
   proves it for every problem."
3. Before: "Is $\mathrm{lse}(\mathbf{x})$ the maximum of ..., and is it
   attained at the softmax? The cell below draws five random logits, ..."
   After: "For five random logits, here is the objective
   $\mathbf{p}^\top\mathbf{x} + H(\mathbf{p})$ at the softmax and at the best
   of a thousand random distributions."
4. Before: "Does the dual solution give a separating hyperplane, are its
   positive multipliers on the margin, and does the dual value meet the
   primal value? The cell below runs the ascent on eight points in the plane
   and checks all three."
   After: "At its limit three things should hold: the hyperplane separates
   the data, the positive multipliers sit on points of margin $1$, and the
   dual value meets the primal value. The cell runs the ascent on eight
   points in the plane."
5. Before: "Does a grid computation over the domain confirm the gap? The cell
   below evaluates $p^\star$ and $q$ on grids and plots the dual function
   against $p^\star$." / "The tent-shaped dual function peaks at
   $-\tfrac12$, ..."
   After: "The cell evaluates $p^\star$ and $q$ on grids and plots the dual
   function against $p^\star$, the counterpart of the first plot of this
   section." / "Where the curve of the clipping problem touched its line, the
   tent peaks at $-\tfrac12$, ..."
6. Before: "We can now bound a constrained optimum from below with any
   nonnegative multipliers, decide from convexity ..., and derive the dual of
   a quadratic program by hand. Two questions remain open."
   After: "Two questions remain open."

The weight-decay cell keeps its question (the review asks for that), and the
water-filling cell keeps its plain statement.

## Repetition

Removed: the sentence repeated forty lines apart (DUA-3); the second
"$q(3) = 7.875$ proves ..." after weak duality; the third derivation of the
clipping curve (DUA-9); one of two water-filling kinship remarks (DUA-13);
"can be read off a table of conjugates" said twice; "softmax maximizes
$\mathbf{p}^\top\mathbf{x} + H$" said twice in one paragraph; "duals of
nonconvex problems ... need not be tight" directly before the closing says
it; the recap sentence (DUA-16); the offset-folding passage cut to three
sentences (DUA-12).

Kept because this file owns them: the multiplier as a price with the worked
derivative, and "a multiplier depends on how its constraint is written". The
closest-point paragraph now carries the numbers ($\nu^\star = -0.2$; $b$ from
$5$ to $5.1$ raises the value by about $0.02$; $0.5202 - 0.5$) that DUA-10
expects the constrained section to drop. Maximum entropy (owner: chapter 41)
is now one clause with :numref:`subsec_mdl-max-entropy-derivation`.

## Ranking, propositions, labels

No proposition demoted (five, all cited or central). The opening now names
the three results that carry the section and says the conjugate can wait for
a second reading; the end of "The Dual Problem" points to
`subsec_mdl-strong-duality` and says the conjugate is not needed for it
(the repair available without moving the conjugate, see deferred). DUA-4:
one sentence after weak duality says the KKT sufficiency proof was this
chain at the KKT multipliers. No labels added. No split markers.

"The previous section" (DUA-2): all eight replaced by
`sec_mdl-constrained-optimization-duality`, `subsec_mdl-kkt-conditions` or
`subsec_mdl-closest-point`. "Thus" after definitions (DUA-15): all four gone.
DUA-8: the $*$/$\star$ convention moved to the line that defines $f^*$.

## Cells to cut or redesign on the development machine

- `mdl-lagrangian-duality-a-lower-bound-from-one-multiplier` (DUA-3): the
  printed values repeat the hand computation; keep the plot, drop the prints.
- `mdl-lagrangian-duality-log-sum-exp-and-entropy` (DUA-14): confirms
  Fenchel-Young by sampling; prose cut to one sentence each side.
- `constrained-duality-gap` (DUA-14): confirms a hand computation; the plot
  is its value.
- Imports cell (DUA-18): `torch`, `nn`, `F`, `math` and the other tabs'
  framework imports are unused.

## Other deferred findings

- **Structural 1** (move "The Convex Conjugate" after "Multipliers as
  Prices"): would move the log-sum-exp cell past the weight-decay cell, which
  this pass may not do. Pointer added instead.
- **DUA-1**: the clipped point is $\mathbf{x}_0$ here and $\mathbf{y}$ in
  Constrained Optimization. Logged, not changed, by instruction.
- **DUA-5 / DUA-17**: same numbers for the half-plane, the closest-point
  example and exercise 6(b); needs the slide changed too.
- **DUA-6**, exercise 2 (a), (b) overlap chapter 41: exercises left alone.
- Exercise 11 should cite the passage of the constrained section it re-solves;
  the Moreau exercise to be received from there. Stale exercise labels.
- Notation load listed in the review ($\lambda$, $p$, $\theta$, $\nu$): only
  $\rho \to r$ done.

## Claims added or changed

1. KKT sufficiency in `subsec_mdl-kkt-conditions` is the weak-duality chain at
   the KKT multipliers, tight by complementary slackness (read against that
   proof).
2. The lower edge of $\mathcal{A}$ is the graph of $p^\star(u)$; the
   shadow-price inequality is its supporting line.
3. $p^\star(u) = \tfrac12(5 - \sqrt{1+2u})^2$, $r = \sqrt{1+2u}$;
   $\mathrm{d}p^\star/\mathrm{d}r$ is $r$ times the price per unit of $u$.
4. Closest point with $(3,4)$, $5$: $\nu^\star = -0.2$, value up by about
   $0.02$ for $b = 5.1$.
5. Penalty and constraint correspond for any convex penalty $\Omega$ (convex
   $\ell$ needed only from constraint to penalty); lasso named.
6. Affine constraints: Slater needs only feasibility (SVM, exercise 3).
7. "Relative interior of $\mathcal{D}$" in the strong-duality hypotheses.
8. The lse cell's shortfall "changes with the draw" (unseeded generator;
   stored value 0.0168).
9. Water-filling as "the standard example of a multiplier with units an
   engineer cares about" (a judgment, from DUA-13).

Numbers quoted in prose that no cell prints, kept after recomputation: $6.43$
and the bound $6$; $0.02$; the review had already recomputed the rest.

## Slides now disagreeing with the body

- Slide "Slater" line states the strict form ($g_i(\bar{\mathbf{x}}) < 0$ for
  all $i$); the body states strictness only for non-affine $g_i$ (it differed
  before this pass too).
- A slide writes $p^\star(s) = \tfrac12(5 - \sqrt{2s})^2$, slope $-4$ at
  $s = \tfrac12$; the body now writes the same curve in $u$
  ($s = \tfrac12 + u$).
