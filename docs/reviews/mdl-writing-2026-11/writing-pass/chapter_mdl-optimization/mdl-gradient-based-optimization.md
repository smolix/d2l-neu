# Writing pass: Gradient Descent and Curvature

File: `chapter_mdl-optimization/mdl-gradient-based-optimization.md`. Prose
only; no code cell, cell ID, figure include or slide changed. Checker:
"OK: only prose changed", no em-dash warning; `tools/lint_source.py` clean.

## Words

Prose words by the checker's count (body and exercises, code excluded):
9,557 before, 9,525 after (-32). The exercises (1,815) are untouched, so the
body went from 7,742 to 7,710. The cuts (edge of stability -108, rates
subsection -124, BFGS -53, the "You can now" paragraph -56, heavy-ball figure
reading, the restated figure descriptions) were nearly used up by the
correctness repairs in "Backtracking Line Search" (+70), the ranking
sentences in the opening and at the seam (+120), the units remark and
dimension clause under Nesterov, and the two-group Summary (+39). The real
shortening waits for the cell cuts listed below.

## Correctness changes

- **GD-4** (verified numerically and by hand). Said: "At $x_0 = 3$ the local
  curvature is $f'' = 27$ and the local stability ceiling is
  $2/27 \approx 0.074$". Now: the update $x \mapsto x(1 - \eta x^2)$ shrinks
  $|x|$ if and only if $\eta x^2 < 2$, so from $x_0 = 3$ a fixed step
  converges for $\eta < 2/9$ and diverges above; the threshold is not
  $2/f''(x_0)$ because a step toward the origin meets the average of $f''$
  on $[0, x]$, which is $x^2$; the descent lemma certifies only $2/27$, a
  third of the safe range. Verification: ran $\eta \in \{0.074, 0.1, 0.2,
  0.22\}$ (converge), $2/9$ (flips between $\pm 3$), $0.223, 0.25, 0.3$
  (diverge; $|x_6| = 6.52\times10^{103}$ at $0.3$ as stored). Agrees with
  exercise 4(b).
  Also replaced "The accepted step roughly tracks $2(1-c)/f''(x)$" and "fails
  for every $\eta$ above about $0.05$": the Armijo test with $c = \tfrac12$
  on $\tfrac14x^4$ reduces to $t^3 - 4t^2 + 6t - 2 \le 0$ in $t = \eta x^2$,
  a monotone cubic with single real root $0.4563$, so it holds if and only
  if $\eta x^2 \le 0.456$ ($\eta \le 0.0507$ at $x_0$); the accepted steps in
  the stored run have $\eta x^2 = 0.28$ to $0.29$ until the cap $\eta_0 = 1$,
  hence "each close to $0.3/x^2$". The parenthetical lemma guarantee
  $2(1-c)/27$ was dropped as redundant.
- **GD-5**. "Training rarely backtracks, because each probe needs another
  forward pass. Learning-rate schedules are the cheaper substitute." now
  gives the noise of a minibatch loss as the first reason and says a
  schedule fixes the step in advance (`sec_scheduler`).
- **GD-7**. "Raising $\eta$ lowers the left endpoint's value and raises the
  right one's" now carries "Once $\eta$ exceeds $1/\lambda_{\max}$".
- **GD-8**. "The per-mode recursion explains why the sharpness stops at the
  ceiling" is gone. The text now says the quadratic model explains the
  oscillation (and single-step loss rises) and that the return of the
  sharpness is outside the model: the third-order Taylor term turns the
  oscillation into a drift toward lower sharpness
  (`Damian.Nichani.Lee.2023`, the citation chapter 9 uses). Agrees with
  `optimization-intro.md` and this chapter's Further Reading.
- **GD-12**. Nesterov's class guarantee is now stated as
  $f(\mathbf{x}_k) - f^\star \le C(1 - 1/\sqrt\kappa)^k$ with the remark
  that this is the gap, so half the exponent of the proposition's distance
  bound on a quadratic.
- **GD-13**. Lower bound: added "as long as the dimension is large compared
  with the number of steps".
- **GD-14**. Gradient descent's iterates "change under every $\mathbf{T}$
  that is not orthogonal; a multiple of an orthogonal matrix changes only
  the effective step" (checked: mapped iterates follow
  $\mathbf{x} - \eta\mathbf{T}\mathbf{T}^\top\nabla f$).
- **GD-10** (the disagreement with Convexity about the factor four): the
  "same units" passage with 13,800 against 3,454 steps is removed here.
  Convexity owns the comparison (chapter review, "Tell the rates once").
- No finding I disagreed with. GD-4's suggested wording "is not much larger"
  I replaced by the factor (three).

## Numbers quoted in prose that no cell prints (all recomputed)

$2/9$; the Armijo threshold $0.456$ and $0.05$; "$0.3/x^2$"; the valley
figure's matrix $\mathrm{diag}(1,20)$ and step $0.097$ (read from
`tools/gen_mdl_optimization_figures.py`; factors $0.903$, $-0.94$ match the
caption); heavy-ball figure numbers (interval $[0.60, 1.50]$ for
$\beta = 0.05$, real root $0.81$, $\beta \approx 0.33$ rate $0.57$,
$\sqrt{0.85} = 0.92$, modes at $\eta\lambda = 0.18, 1.82$); $691$ steps as
$\log 10^6/\log(101/99)$; the kept originals ($69$, $6{,}908$, Nesterov
rates, secant errors, trust-region step $\tau = 1.60$,
$\mathbf{p} = (-0.19, 0.46)$, predicted decrease $0.316$, $L \approx 0.4$).

## Six before/after pairs

1. Before: "This section asks how far a step along the gradient can be
   trusted, and how many steps that limit forces on us."
   After: "How far can a step along the gradient be trusted, and how many
   steps does that limit force on us?"
2. Before: "The cell sweeps $\eta$ across the ceiling on
   $\mathbf{A} = \mathrm{diag}(1, 10)$, where $L = 10$ and the ceiling is
   $\eta = 0.2$, and prints both factors, the spectral radius, and the error
   after $60$ steps from $(1, 1)$."
   After: "On $\mathbf{A} = \mathrm{diag}(1, 10)$ the ceiling is
   $\eta = 0.2$. Before reading the sweep below, guess which step leaves the
   smallest error after $60$ steps from $(1, 1)$: the cautious $0.02$, the
   step $0.1 = 1/L$ that the descent lemma favoured, or one closer to the
   ceiling."
3. Before: "The per-mode recursion explains why the sharpness stops at the
   ceiling."
   After: "It explains half."
4. Before: "The cell runs Newton's method and gradient descent with
   $\eta = 1$ from this point."
   After: "Which way will Newton's step move $y$ from here, and which way a
   plain gradient step with $\eta = 1$?"
5. Before: "The cell counts the steps each method needs for a $10^{-6}$
   relative error on $\mathrm{diag}(1, \kappa)$ from $(1, 1)$ as $\kappa$
   grows."
   After: "It remains to collect the counts that the opening promised. On
   $\mathrm{diag}(1, \kappa)$ from $(1, 1)$, six digits should cost gradient
   descent $\tfrac{\kappa}{2}\log 10^6 \approx 6.9\,\kappa$ steps and heavy
   ball somewhat more than $6.9\sqrt\kappa$."
6. Before: "Among all symmetric matrices that satisfy the secant equation,
   it is also the closest to $\mathbf{M}_k$ in a weighted Frobenius norm,
   but we do not use that property."
   After: "**BFGS** (Broyden, Fletcher, Goldfarb, and Shanno) fills in the
   rest by changing $\mathbf{M}_k$ as little as possible: in a suitably
   weighted Frobenius norm, the symmetric matrix closest to $\mathbf{M}_k$
   that satisfies the secant equation is"

## Repetition

Removed or reduced:
- Rates table and the definition of strong convexity (owner: Convexity,
  `subsec_mdl-convergence-rates`). "From Quadratics to Convex Functions" is
  now a bridge of two short paragraphs. The label
  `eq_mdl-opt-gd-rate-strongly-convex` had to stay, so the strongly convex
  rate is kept as one display under it; the other two rates are a clause
  each, and the definition is a reference to `eq_mdl-opt-strong-convexity`.
- Edge-of-stability experiment (owner: chapter 9). Cells kept, prose cut
  from about 480 to 370 words, `sec_optimization-intro` cited for the
  experiment and its discussion.
- Newton against chapter 9: `subsec_gd-newton` cited for the step and for
  the cost; the "$10^8$ parameters, $10^{16}$ numbers" sentence is gone. The
  two labelled displays, affine invariance, the convergence proof, damping
  and the saddle stay (this file owns them; chapter 39 cites
  `subsec_mdl-why-not-newton` for damping and cost).
- Logistic smoothness constant: given once, in the Newton subsection where
  it is used (GD-2), no longer in the definition of $L$.
- Heavy-ball figure numbers: given once, after the proposition that yields
  them (GD-11); the paragraph before the figure describes the regimes in
  words.
- The running example's formula appears once (opening) instead of three
  times; restated figure descriptions under `fig_mdl-opt-eta-tent` and
  `fig_mdl-opt-hb-eigenvalues` shortened; the restatement of the guaranteed
  decrease after the descent-lemma figure reduced to a sentence.
- Double ending: the "You can now" paragraph is deleted (GD-19). Its one
  item the Summary lacked (exact gradients, stationary point only) is the
  open-questions sentence of the Summary.

Kept because this file owns them: descent lemma, stationarity bound,
stability ceiling, optimal step, the counts 69 and 6,908, heavy-ball and
Nesterov propositions, Newton's quadratic convergence.

## Ranking; Proposition blocks; labels

- No Proposition block demoted: every block here is cited by an exercise,
  by another section or by chapter 9. The BFGS bullet "proofs" became
  running prose (the two checks remain, exercise 11(d) refers to them).
- Ranking added: the opening names the three results to remember (descent
  lemma, optimal step, $\sqrt\kappa$ law) and says the last two parts are
  used far less in deep learning; the paragraph after the seam names
  quadratic convergence as the result to remember and says BFGS can be
  skipped on a first reading.
- Label added: `subsec_mdl-edge-of-stability` on "The Edge of Stability"
  (asked for by the chapter review so that Adaptive Methods and Further
  Reading have a target).

## Split

Marker at line 983, before `## Newton and Quasi-Newton Methods`:
`<!-- split: new section "Curvature Methods: Newton, Quasi-Newton, and Trust Regions" starts here -->`

- Part one now ends on its own argument (momentum costs no extra gradients;
  gradients alone cannot do better in high dimension; removing $\kappa$
  needs curvature).
- The paragraph after the seam poses part two's questions and cites part
  one by label only.
- Backward dependencies removed from part one: the logistic-loss formula
  ("of the Newton subsection below") and "curvature models" in the roadmap.
  Remaining forward mentions in part one, to delete at the split: the
  opening's "Newton's method needs a single step if we can afford its
  Hessian" and the two roadmap sentences beginning "The last two parts".
- Part two reuses from part one: `eq_mdl-opt-armijo` (by label), the saddle
  $\tfrac12x^2 + \cos y$ (restated), and Python names defined in part-one
  cells (`np`, `d2l` only; `f_s`, `grad_s`, `hess_s`, `bfgs_update` are
  defined in part two). The imports cell must be duplicated.
- Summary is in two labelled groups.
- Exercises to move with part two: 5 (reparametrization, Newton), 11
  (BFGS), 12 (trust region); exercise 1(b) and 2(b, c) mention Newton's
  direction and the Newton data cell and need a cross-reference.

## Cells to cut or redesign on the development machine

- `mdl-gradient-based-optimization-the-edge-of-stability-1` and
  `gradient-based-optimization-edge-of-stability` (GD-9): chapter 9's
  experiment in NumPy; cut both, keep the two prose paragraphs and point
  exercise 10 at chapter 9's cell.
- `mdl-gradient-bfgs-quadratic` (GD-16, GD-17): confirms finite termination
  in six dimensions; its plot shows nothing the two printed lines do not.
  Cut, or drop the `d2l.plot` call.
- `gradient-based-optimization-contraction` (no finding ID): confirms
  $\rho(\eta^\star)$; could be one more printed column of the sweep cell.
- `gradient-based-optimization-steepest-direction` (no finding ID): confirms
  the Cauchy-Schwarz proposition by a scan; prose shortened.
- `gradient-based-optimization-newton` (GD structural): uses
  $\mathrm{diag}(1, 100)$; the review asks for $\mathrm{diag}(1, 1000)$
  (6,908 steps) to reduce the number of test quadratics.

## Other deferred findings

- GD-3: $\nu$ kept (the code variable is `nu`, exercise 2 uses it); a clause
  now says why. Clash with the chapter's notation table, where $\nu$ is an
  equality multiplier.
- GD-6: figure captions not touched; the valley's matrix and step are now
  given in the prose. Redrawing the panel on $\mathrm{diag}(1, 10)$ is a
  generator change.
- GD-9: see cells.
- GD-16: pruned and motivated, not cut (the label `eq_mdl-opt-bfgs` and
  exercise 11 need the display).
- GD-18: reference added; proposition still stated as "if and only if"
  with only sufficiency proved (said in the text).
- Exercise 12(c) (a drill) and exercise 10 (should point to chapter 9 once
  the cells go): left.
- Four quadratics (structural): unchanged except in prose.

## Claims added or changed

1. A fixed step on $\tfrac14x^4$ from $x_0 = 3$ converges for $\eta < 2/9$
   and diverges above it; the average of $f''$ on $[0, x]$ is $x^2$; the
   descent lemma certifies a third of the safe range.
2. Armijo with $c = \tfrac12$ on the quartic holds iff $\eta x^2 \le 0.456$;
   accepted steps are close to $0.3/x^2$.
3. Along the opening gradient the curvature is nearly $L$
   ($1004/104 = 9.65$), which is why the lemma's bounds are nearly tight
   there.
4. At $\eta = 0.01$ the actual changes equal $\eta$ times the slopes to
   within one percent (stored output: 0.3 to 0.5 percent).
5. $\eta = 2/11$ "wins by two orders of magnitude" in the sweep
   ($8.3\times10^{-6}$ against $1.8\times10^{-3}$).
6. Edge of stability: the quadratic model explains the oscillation, not the
   return of the sharpness (third-order term, Damian et al.).
7. Strongly convex rate displayed alone; the convex $O(1/k)$ rate and the
   stationarity bound in a clause each; Convexity "also compares these
   bounds with what the quadratic achieves" (true of the current Convexity
   text; depends on that file keeping its rate check).
8. Nesterov's class bound is for the gap, half the exponent of the
   quadratic's distance bound; lower bound needs large dimension.
9. "By the lower bound, gradients alone cannot do better in high dimension."
10. BFGS: the right factor annihilates $\mathbf{y}$ and fixes vectors
    orthogonal to $\mathbf{s}$; the update is the closest symmetric secant
    matrix in a weighted Frobenius norm (Nocedal and Wright); the curvature
    condition holds for every step on a strictly convex function and is
    otherwise enforced by the Wolfe condition (the derivation of that was
    cut).
11. Part two's opening judgment: for training networks these answers matter
    less, because the Hessian is too large to form and minibatch noise
    spoils curvature estimates from gradient differences.
12. Training rarely backtracks because a minibatch loss is too noisy to
    compare two steps.

## Slides that now disagree with the body

- Backtracking slide: "Fixed $0.3$ diverges: the ceiling at $x_0$ is
  $2/27=0.074$." Wrong (GD-4); the threshold from $x_0 = 3$ is $2/9$.
- Any slide that reads the edge-of-stability cell as explained by the
  per-mode recursion should be checked against the new wording (I found no
  such sentence by grep, only the cell placeholders).
