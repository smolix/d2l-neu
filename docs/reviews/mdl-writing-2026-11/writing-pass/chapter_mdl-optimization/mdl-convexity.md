# Writing pass log: Convexity

File: `chapter_mdl-optimization/mdl-convexity.md`. Reviews followed:
`40-optimization/mdl-convexity.md` (CV-1 to CV-18) and `40-optimization.md`.

## Words

Checker count (prose with exercises): 9,338 before, 9,225 after (-113).
Body without exercises: about 7,320 before, 7,210 after. The cut is smaller
than the review's estimate because CV-12, CV-14 and CV-15 each needed added
sentences, and because the section could not be reordered around its cells.
Checker: "OK: only prose changed", no em dash warning; `lint_source.py`
clean. No split markers (section is not to be split).

## Structure

- **Payoff brought forward.** The opening now states "every local minimum is
  a global minimum", says the proof is three lines, and points to
  `subsec_mdl-why-convexity-matters`. It also ranks the section's results
  (tangent bound, local = global, linear rate with the PL inequality) and
  gives a short path for a reader in a hurry.
- **One prose reorder (CV-9, structural finding on Jensen).** The heading
  `## Consequences of Convexity` with its label and introduction moved up to
  stand before `### Jensen's Inequality`, so that Jensen is the first of the
  three consequences the introduction promises. No cell moved relative to
  another; no heading text or label changed. "A Numerical Check of the Three
  Tests" is now the last subsection of "Convex Functions".
- **Not done (needs cells to move):** the local-global proposition directly
  after the chord condition with the double-well cell; the three-tests check
  directly after its proposition (CV-6); the $x^2$ against $x^4$ cell next
  to the linear rate. The proposition could have moved alone, but it would
  then sit in a different subsection from its figure, its cell and the label
  other text cites, so I left it and deferred.

## Correctness changes

No review finding claimed a mathematical error. Changes of substance:

- **CV-12.** Was: "$(1 - 10^{-3})^k \le 10^{-3}$ requires $k \approx 6{,}900$
  steps", later "about $3{,}450$ steps for a thousandfold reduction", and
  6,900 again in the closing. Now: six digits of the gap, bound about
  13,800, squared factor about 6,900, best fixed step $2/(\mu+L)$ 3,454,
  stated once after the rate check, with "One factor of two belongs to the
  proof and one to the step." Recomputed: 13,809, 6,904, 3,454. Note: 6,900
  is the count for the factor $(1-\mu/L)^2$, not a run from a particular
  start (from $(1,1)$ the stiff coordinate dies in one step and the relative
  count is smaller), so the text says "at the squared factor".
- **CV-10.** "The proposition has three corollaries" is now "Three more
  facts of the same kind have one-line proofs". The small-gradient remark
  now carries its hypothesis ("provided that the iterates stay within a
  bounded distance of $\mathbf{x}^\star$") with a forward reference to the
  rate proof, which shows it.
- **CV-11.** The statement of the uniqueness proposition is unchanged
  (exercise 3 quotes it). The existence proof is now given for
  differentiable $f$ on $\mathbb{R}^d$ from `eq_mdl-opt-strong-convexity`,
  and says in one clause how the general case runs (subgradient of
  $f - \tfrac\mu2\|\cdot\|^2$ at an interior point, the old proof). Checked:
  the right side of the strong-convexity bound exceeds $f(\mathbf{x}_0)$ when
  $\|\mathbf{y}-\mathbf{x}_0\| > 2\|\nabla f(\mathbf{x}_0)\|/\mu$
  (Cauchy-Schwarz). The boundary counterexample
  ($x + \tfrac12x^2$ on $(0,1]$) is cut; it could not go to an exercise in
  this pass. One display equation went with the old proof (60 to 58
  delimiters).
- **CV-14.** The PL cell's step $1/(2L)$ now has its reason (at $1/L$ the
  local factor $1 - f''(0)\eta$ is 0) and the one-line origin of
  $1 - 3\mu/(4L)$ (decrease $\eta(1 - L\eta/2)f'^2 = \tfrac{3}{8L}f'^2$).
  Checked against the stored output (0.9835).
- **CV-17.** Summary: "the PL inequality keeps the linear rate" is now
  "Where a PL inequality holds, the linear rate survives without convexity".
- **CV-5.** The parenthesis on closed and lower-dimensional domains is one
  sentence after the examples; the clause on closed full-dimensional domains
  is gone.
- **CV-2, CV-4.** Wording fixed as proposed.
- Disagreed with: none. Exercise 11(d) kept (duplication with chapter 9 is
  small, and the brief limits exercise edits to errors).

## Before and after (verbatim)

1. Before: "Do the three tests agree on a concrete loss? The cell checks
   them on the least-squares loss". After: "We run the three tests on the
   least-squares loss ... The outcome is known in advance, and more than its
   sign: expanding the squares as we did for $x^2$ gives the chord gap".
2. Before: "Does the measured contraction obey the bound, and by how much?"
   After: "The bound must hold in every run; a measurement can tell us how
   much room it leaves."
3. Before: "Does the covariance reading hold numerically? The cell computes
   the eigenvalues". After: "The cell confirms the three facts at six random
   logits".
4. Before: "How large is the gap in a concrete case, and is the bound $0$
   for the KL divergence ever reached?" After: "as $q$ moves toward $p$
   along the mixture ..., does the divergence fall in proportion to
   $\varepsilon$, or faster?"
5. Before: "We now return to the question of the opening: what does the end
   of a run mean? For convex functions there are three answers." After:
   "Convexity has three consequences that the rest of the book uses, and all
   come from one source, a line or plane that bounds the function from below
   everywhere."
6. Before: "Functions with this property are called *convex*." After:
   "Functions whose chords all lie on or above the graph are called
   *convex*, and for them the question has the best answer we could hope
   for: every local minimum is a global minimum."

## Repetition

Removed: the second telling of the opening experiment (CV-1: the opening now
gives the phenomenon and one number, the numbers stay at the cell); the
recap paragraph before the Summary (CV-16; its count
$\kappa\log(1/\epsilon)$ went into the Summary, its second half is the
ending of "Implicit Bias"); the paragraph "Which characterization to use",
which repeated the three subsection openings; the forward sentence in "The
Chord Condition"; the figure walk-throughs that repeated captions; the
paragraph on the two senses of $\succeq$ (a clause at each first use);
polyhedra, affine images and the two descriptions of the hull (CV-3); the
claim that the previous section "stated two rates" (the chapter review makes
this section the home of the rates, so the text no longer depends on what
the first section keeps).

Kept because this file owns them: subgradients and the subdifferential
(definition now self-contained, no longer leaning on the calculus chapter);
the rates and their proofs; the proof slack against the quadratic; strong
convexity; the permutation-symmetry argument; Jensen's inequality.

## Ranking

Demoted from Proposition blocks to a sentence with its justification:
"intersections preserve convexity", "sublevel sets are convex", "supporting
lines in one dimension" (CV-8: the 230-word proof is now about 100 words of
running text with the three-slope inequality displayed; nothing cites it).
Kept as blocks: three characterizations, Jensen, KL corollary, local =
global, uniqueness, the two rates, log-sum-exp, PL. Said in the text: which
three results to remember; that the tangent bound is the test to remember;
which part of the equivalence proof to read; that Jensen can be skipped by a
reader interested only in optimization. Labels added: none.

## Cells to cut or redesign on the development machine

- `convexity-three-lenses` (CV-6): cannot fail. Run the tests on $\phi$ or
  $(ab-1)^2$ so that sampling refutes something, or check the closed-form
  gaps to rounding; then move it up to follow the proposition.
- `convexity-jensen-mc` (CV-7): drop the AM-GM and KL bulk lines; the prose
  now says they "can only print `True`".
- `convexity-lse-hessian` (CV-13): confirms a derivative; replace by the
  largest eigenvalue against the bound $\tfrac12$ of exercise 6(b).
- `convexity-rate-check` (structural finding): keep, but its last print
  ("bound holds at every step") is a theorem check.
- `convexity-pl-rate` (CV-14): consider $\eta = 1/L$ with a different
  example, or keep with the reason now given.
- `convexity-basins` (CV-1): could move into the opening.

## Other deferred findings

- Full reorder of the first half (structural finding): needs cells to move.
- A Jensen exercise on a loss; dropping exercise 11(d); moving the
  one-dimensional supporting-line proof and the boundary counterexample into
  exercises: no exercise changes in this pass.
- TensorFlow, JAX and MXNet outputs not checked.

## Claims added or changed

- Convex hull: "It is convex, and it is the smallest convex set that
  contains $S$" (standard; the exercise proves the inclusion).
- Tangent-bound proof remark: the choice of the intermediate point returns
  in Jensen's proof (the mean).
- $|y| \ge gy$ for $g \in [-1,1]$.
- Chord and tangent gaps of least squares stated before the cell (same
  formulas as before; rechecked numerically).
- KL along the mixture: "Equality is approached quadratically".
- Existence proof as above; "which no loss in this book does" (jump at a
  domain boundary).
- Step counts 13,800, 6,900, 3,454 (recomputed).
- PL: decrease $\tfrac{3}{8L}f'(x_k)^2$ at $\eta = 1/(2L)$; "peaks at
  $L = 8$".
- Implicit bias (CV-15): overparameterized networks fit the data in many
  ways with different predictions elsewhere; a fixed diagonal preconditioner
  $\mathbf{D}$ reaches the interpolant minimizing
  $\mathbf{w}^\top\mathbf{D}^{-1}\mathbf{w}$ (verified numerically on a
  $4\times10$ system, agreement to $10^{-15}$), "so the per-coordinate steps
  of Adam need not select the solution that gradient descent selects".
- Numbers quoted that no cell prints and that I kept: $\sqrt{e} \approx
  1.649$ (printed), 0.458 (printed), the step counts above (recomputed).

## Slides that now disagree with the body

- "Check: How Many Steps on the $\kappa = 10^3$ Quadratic?" and the wrap-up
  use a thousandfold reduction (6,900 by the bound, 3,450 on the quadratic);
  the body uses six digits (13,800, 6,900, 3,454). Both are correct, but
  6,900 means different things in the two places.
- "At a Kink": mentions $[f'_-, f'_+]$; still in the body, no longer a
  proposition.
- Wrap-up and the opening slide quote "46% expected" and $x \approx 0.93$
  for the opening; the body now gives those numbers only at the cell.
- Slide order has Jensen under "Convex functions" before "Consequences";
  the body now files Jensen under Consequences.

## Notation

No shared symbol changed. $\mu$ remains the strong-convexity and PL constant
(clash with the Lagrange multiplier of chapter 39, noted in the chapter
review). $\phi$ is the double well here and is reused in a duality exercise
that cites `subsec_mdl-recognizing-convexity`.
