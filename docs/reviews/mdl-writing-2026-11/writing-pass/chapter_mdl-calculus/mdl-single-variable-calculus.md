# Writing-pass log: Single Variable Calculus

Source: `chapter_mdl-calculus/mdl-single-variable-calculus.md`. Prose only; no
code cell, figure include, existing label or slide changed
(`check_mdl_prose.py`: "OK: only prose changed"; `lint_source.py` clean).

## Words

Checker count (prose, captions and exercises to the slides marker): 8244
before, 7966 after (-278). Body without exercises: about 6100 to 5900;
exercises 2143 to 2070. The review asked for 800 to 1000 fewer. The rest is
blocked by the rules of this pass: the two cells to cut (SVC-4, SVC-10) and
the two figures to drop (secant, smooth-not-analytic) must stay and need a
sentence each, and SVC-5, SVC-7 and SVC-19(c) add text.

## Correctness changes

- **SVC-1.** Was: "The error has a simple form. It equals
  $\tfrac12\cdot\tfrac14\cdot0.1^2$". Now: "To the digits shown it equals".
  Recomputed: gap = 0.0012494795. The stored output prints gap/eps^2 = 0.1249
  at eps = 0.1; the text after the cell now says so and
  the Lagrange subsection explains it ($\tfrac18-\epsilon^2/192=0.12495$,
  from $P_4$; recomputed).
- **SVC-5.** The lead of "Computing Derivatives" now says which table entries
  are derived (exponential, logarithm, power rule) and which are taken from a
  first course (sine, cosine, existence of a base with slope 1 at 0). Added
  after the inverse-function rule: $x^n=e^{n\log x}$ gives $nx^{n-1}$ for
  real $n$, $x>0$ (checked by hand).
- **SVC-6.** "The inverse-function rule below identifies $c_a$" is now "Once
  the logarithm is available, the chain rule identifies $c_a$", in both
  places.
- **SVC-7.** Fermat's lemma is stated and proved before the Mean Value
  Theorem (the four lines that were inside the Rolle argument, for a local
  extremum). The Rolle step and the later remark on $f'(x)=0$ cite it. The
  lead no longer says "We have been reading the derivative as if ...".
- **SVC-15.** The iterates from $x_0=1.3$ are marked "(not printed)".
  Recomputed: $-1.162, 0.859, -0.374, 0.034$; boundary near 1.39 (1.39
  converges, 1.40 diverges).
- **SVC-18.** The softplus slopes are now quoted at $x=\pm0.1$ for
  $\beta=4$ (0.40, 0.60) and $\beta=16$ (0.17, 0.83), both drawn in the
  figure (recomputed: 0.401/0.599, 0.168/0.832).
- **SVC-19(c), exercise 12(e).** The exercise cited a claim the text does not
  make; it now asks to compare with what the subsection reports.
- **Exercise 9(a).** Added "three times continuously differentiable".
- Chain-rule proof: "Thus for every $t$" is now "For every $t$, including
  $t=0$" (the identity holds at $t=0$ by $\rho(0)=0$).
- No finding disputed.

## Six before/after pairs

1. Before: "The smallest interesting case is a polynomial whose quotient we
   can also expand by hand. Take $f(x) = x^2 + 1701(x-4)^3$ at $x = 4$. Does
   the secant slope settle as $\epsilon$ shrinks?"
   After: "How small must $\epsilon$ be before the secant slope can be
   trusted? That depends on the function. We choose a polynomial with a large
   cubic coefficient, $f(x) = x^2 + 1701(x-4)^3$, whose slope at $x=4$ is
   $8$, and print the secant slope there for shrinking $\epsilon$."
2. Before: "Automatic differentiation applies these same rules mechanically.
   Does it reproduce the hand results $f'(4)=8$ for the cubic and
   $s'(0)=\sigma(0)=\tfrac12$?"
   After: "Automatic differentiation applies the same rules mechanically, so
   it returns the hand results $f'(4)=8$ for the cubic and
   $s'(0)=\sigma(0)=\tfrac12$:"
3. Before: "Do ten steps from $x_0=1$ show these regimes? The next cell runs
   the recursion for five step sizes, prints the tenth iterate, and plots the
   trajectories."
   After: "Ten steps from $x_0=1$ at five step sizes show one run per
   regime."
4. Before: "The proofs use one fact."
   After: "The sum and product rules are routine. Only the chain rule needs
   care: the obvious argument divides by $h(x+\epsilon)-h(x)$, which may be
   zero. The proofs use one fact."
5. Before: "The Mean Value Theorem gives an exact expression for the error of
   $P_n$, with one unknown point."
   After: the same sentence, followed by "This is the main result of the
   section: the descent lemma and Newton's rate are both its case $n=1$."
6. Before: "We can now predict how a function changes from its slope and
   bound the error of that prediction from its curvature, choose a step size
   that a curvature bound certifies, run Newton's method and recognize when
   it fails, and say what a slope means at a corner. A network, however, has
   millions of weights that move together."
   After: "All of this was for one weight. A network has millions of weights
   that move together."

## Repetition

Kept, because this file owns them: the linear approximation with its
converse, the proofs of the rules, the Mean Value Theorem, the Lagrange
remainder, the one-variable descent lemma (now with a pointer to
:eqref:`eq_mdl-opt-descent-lemma` for the many-variable form).

Reduced to a clause or a few sentences with a reference:

- Against `sec_calculus`: the difference quotient (display removed, one
  paragraph citing the Basics), $|x|$ at $0$ in the derivative subsection (a
  clause pointing to `sec_mdl-tangent-fails`), the rectangle picture of the
  product rule (deleted), "rates multiply" example (deleted), the two
  remarks on autodiff (one sentence each).
- Against chapter 40, Newton: the Proposition and Proof block "Newton's
  error recursion" is now a three-sentence derivation ending in the labelled
  equation `eq_mdl-newton-error` (label kept; exercise 4 and the slides use
  the subsection), with the theorem referred to
  :eqref:`eq_mdl-opt-newton-error`. Damping is one sentence with
  :numref:`subsec_mdl-why-not-newton`; the $\alpha=\tfrac12$ number and the
  $\tfrac\pi2x_t^2$ growth claim are gone.
- Against chapter 40, subgradients: definition, subdifferential and the two
  intervals in one paragraph; the optimality condition as a sentence with
  its one-line reason (label `eq_mdl-subopt` kept, exercise 6 cites it);
  reference to :numref:`sec_mdl-convexity`. The worked $g=0.3$ / $g=1.5$
  example is deleted.
- Inside the file: SVC-9 (the $2/f''$ identification is made twice, not
  three times); the real-analytic passage (SVC-12) is three sentences and
  the term "real analytic" is no longer defined; the recap before the
  Summary is gone (SVC-20); the caption paraphrase after the
  second-derivative figure is gone; SVC-16 sentence deleted.
- Exercise 9: parts (c) and (d) (rounding against truncation, the float
  sweep) removed in favour of a pointer to the gradient-check exercise of
  `sec_mdl-multivariable_calculus`, the home in the chapter table. Lost with
  them: the closed forms $\epsilon_+=2\sqrt{u|f|/|f''|}$ and
  $\epsilon_0=(3u|f|/|f'''|)^{1/3}$; if wanted, they belong in that
  exercise (which writes machine epsilon, not unit roundoff $u$).

## Ranking

- Demoted from Proposition blocks: "Proposition (exponentials)" (now a
  sentence with its display), "Proposition (Newton's error recursion)"
  (derivation in running text, label kept), "Proposition (optimality for
  convex functions)" (sentence, label kept).
- New block: "Lemma (Fermat)" with proof (SVC-7).
- Judgements added: the three results that carry the section (opening); "A
  reader who trusts the rules can start at" the Mean Value Theorem; "Only
  the chain rule needs care"; the Lagrange remainder as "the main result of
  the section"; "The only idea is the choice of an auxiliary function"; the
  descent lemma as "the one that optimization uses most".
- Label added: `subsec_mdl-descent-lemma` on "The Descent Lemma" (asked for
  in structural finding 5). Used once in the file.
- Heading reworded (SVC-19a), label unchanged: "Kinks in Stochastic
  Training" is now "Kinks in a Network".

## Split markers

None; the file is not to be split.

## Cells to cut or redesign on the development machine

- `single-variable-calculus-autograd-check` (SVC-4): confirms two hand
  derivatives; cut. Prose is one sentence.
- `single-variable-calculus-taylor-series` (SVC-10): plot only; cut. Prose is
  two sentences.
- `mdl-single-variable-calculus-newton-s-method` (SVC-15): add a second
  start $x_0=1.3$ to the second loop so the convergent run is printed.
- `single-variable-calculus-gradient-descent`: prints $(1-2\eta)^{10}$, which
  the closed form gives; keep only if the plot is wanted (no finding ID).

## Other deferred findings

- SVC-2: the secant figure (same as the Basics one) and SVC-12: the
  $e^{-1/x^2}$ figure. Figure includes cannot change in this pass; both now
  hang on a short paragraph and can go with it.
- SVC-11: best parabola on the softplus instead of the sine needs a new
  figure.
- SVC-13 done; SVC-21: "below"/"above" as cross-references are gone from the
  body (labels are used); "the logistic loss above" remains once in the
  opening.
- Exercise 10 (candidate to trim) and the MIT credit on exercise 4: left
  for the author.
- Chapter 40 has no sentence saying that a subgradient step need not be a
  descent step, so that warning stays here (moved to the end of
  "Subgradients and Optimality", SVC-19b). If the convexity section takes
  it, delete it here.

## Claims added or changed

- gap/eps^2 at 0.1 is $\tfrac18-\epsilon^2/192=0.12495$ (recomputed).
- "the secant slope is 25.11, three times the limit" (stored output; 3.1).
- "$o(\epsilon)$ ... of order $\epsilon^2$ ... the rule for twice
  differentiable functions" (Lagrange remainder, $n=1$).
- "the obvious argument [for the chain rule] divides by
  $h(x+\epsilon)-h(x)$, which may be zero" (standard).
- Fermat's lemma for local extrema; power rule for real exponents.
- "the descent lemma and Newton's rate are both its case $n=1$" (true of the
  two derivations in the file).
- Softplus slopes at $\pm0.1$ (above).
- **Bertoin et al., from memory of the paper's abstract and the figures
  already quoted in exercise 12(b); not checked against the paper:** the two
  conventions for $\mathrm{ReLU}'(0)$ gave different gradients "for every
  initialization in 16-bit arithmetic, for roughly half of them in 32-bit,
  and for none in 64-bit", and the choice "affects training with plain SGD,
  and much less so with batch normalization or Adam". Exercise 12(b) says
  100%, 43%, 0%.
- Numbers no cell prints, kept after recomputation: 0.643147, 0.644397,
  0.00125; $(2^\epsilon-1)/\epsilon=0.69315$ and 2.30259 at $10^{-6}$;
  $P_4(0.5)=0.974072$, $s(0.5)=0.974077$; descent-lemma values 0.4744,
  0.4741, 0.1931, 0.1269; Newton ratios 0.42, 0.64, 0.87, 0.99; the 1.3 run
  and the 1.39 boundary.

## Slides that now disagree with the body

- "Damping (a step $\alpha_t<1$ times as long, chosen so $f$ drops) restores
  convergence": the body now only names damping and refers to chapter 40.
- "At $x=\pm0.01$: $0.49/0.51$ for $\beta=4$, $0.35/0.65$ for $\beta=64$":
  the body quotes $x=\pm0.1$, $\beta=4$ and $16$.
- "the $0.00125$ left over is ... $\tfrac12\cdot\tfrac14\cdot0.1^2$" and
  "the gap $0.00125=\tfrac18\cdot0.1^2$": equal only to the digits shown
  (SVC-1).
- Slide title "... Smooth Is Not Analytic": the body no longer defines
  "analytic".
- Any slide titled after "Kinks in Stochastic Training" (heading renamed).
