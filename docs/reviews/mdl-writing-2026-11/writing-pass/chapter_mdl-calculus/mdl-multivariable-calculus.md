# Writing pass: `chapter_mdl-calculus/mdl-multivariable-calculus.md`

Checker: "OK: only prose changed", no em-dash warning, 2 split markers.
`tools/lint_source.py`: clean. No code cell, cell ID, figure include, or
slide changed. No label removed.

## Words

Prose words by the checker (body and exercises): **8063 before, 8043 after**
(-20). The checker counts every `:label:`, `:numref:` and `:eqref:` as about
three words, and this pass added 4 labels and about 15 references in place of
positional phrases ("later in this section", "the opening"), so the running
prose is shorter than the number says, by roughly 150 to 200 words. It is not
a large cut: the correctness repairs (the adjoint, the two written-out lines
of the Clairaut proof, the notation paragraph) took back about half of what
the cuts freed.

## Correctness changes

- **MVC-12 (adjoint written as a partial derivative).** Said:
  "$\bar a = \frac{\partial f}{\partial a}$ ... the total derivative of the
  output with respect to that node", one page after teaching that
  $\partial f/\partial p = 1$ while the total is $7$. Now: the adjoint
  $\bar n$ of a node $n$ is defined in words as the total derivative
  $\mathrm{d}f/\mathrm{d}n$, the sum over all paths from $n$ to $f$ of the
  products of edge derivatives, and `eq_mdl-adjoint` holds only the seed and
  the recursion, $\bar f = 1$, $\bar n = \sum_{c} \bar c\,\partial c/\partial n$
  (label kept; generic node renamed from $a$ to $n$, since $a$ is a node of
  the example). The convention is stated once, where the chain rule is first
  written: $\partial$ for edge derivatives, upright $\mathrm{d}$ for totals,
  with a parenthesis that for an input the total is the ordinary partial
  derivative of $f$ as a function of the inputs. Applied throughout the
  chain-rule part: the first chain rule, `eq_mdl-chain-rule-paths`, the
  skip-edge sum, the forward-mode display, $\mathrm{d}u/\mathrm{d}w = 160$,
  $\mathrm{d}v/\mathrm{d}w = 64$. Verified: on the skip-edge graph the
  recursion gives $\bar p = \bar r\cdot 2p + \bar f\cdot 1 = 7$ at $p = 3$,
  and this check is now in the text.
- **MVC-18 (Lagrange remainder restated with another hypothesis).** Said: "if
  $\varphi'$ is continuous on $[0,1]$ and $\varphi''$ exists on $(0,1)$,
  then", attributed to the previous section. Now: cites
  `eq_mdl-lagrange` of degree one, as proved there. Its hypothesis (twice
  differentiable on an open interval containing $0$ and $1$) holds because
  $f$ is assumed twice continuously differentiable near the segment.
- **MVC-10.** The heuristic before the Lagrange proposition now opens "Let $f$
  and $g$ be differentiable."
- **MVC-17 (Clairaut proof in one sentence).** The two applications of the
  mean value theorem are written out in one display, with $\tau,\sigma$
  between $0$ and $h$, and `eq_mdl-mvt` is cited. Checked: the second
  application is to $s \mapsto \partial_i f(\mathbf x + \tau\mathbf e_i +
  s\mathbf e_j)$, which needs only that $\partial_j\partial_i f$ exists near
  $\mathbf x$, as the proposition assumes. Distance bound now
  $\sqrt2\,|h|$.
- **MVC-6.** Said: "The learning rate $\eta > 0$ absorbs the length of the
  gradient into the step." Now: "the update does not normalize the direction.
  Its step has length $\eta\|\nabla\ell\|$, which shrinks as the gradient
  does."
- **Backward dependency across the seam (my finding).** The Taylor subsection
  obtained $\varphi'(t)$ and $\varphi''(t)$ from "the chain rule", which after
  the split comes in the later section. It now derives them from the linear
  model `eq_mdl-nabla_use` (the slope of $f$ along $\boldsymbol\delta$),
  applied once to $f$ and once to each first partial, which is differentiable
  because the second partials are continuous.
- **Second-order proposition.** Item 3 now says "whatever the zero
  eigenvalues", and the closing line of the statement names the inconclusive
  case (singular semidefinite $\mathbf H$ falls under neither 2 nor 3). The
  proof is untouched.
- Stored outputs: all quoted numbers agree with
  `outputs/pytorch/.../mdl-multivariable-calculus.json` (gaps $1.786\times
  10^{-4}$, $1.779\times10^{-6}$; $0.000119$, $0.000921$; eigenvalues). The
  factor "about eight" is now "a little less than eight" (the ratio is 7.7).
- No finding disagreed with. No number is quoted that a cell does not print,
  apart from hand arithmetic already in the text.

## Before and after (verbatim)

1. Cell introduction, linear model.
   Before: "Does the linear model predict a small step of the running example,
   and how does its error shrink as the step shrinks? The next cell evaluates
   both sides of :eqref:`eq_mdl-nabla_use` ..."
   After: "The definition promises an error that is
   $o(\|\boldsymbol{\epsilon}\|)$ and nothing more. Before running the next
   cell, guess by what factor the error of the linear model falls when the
   step $\boldsymbol{\epsilon} = (0.01, -0.03)$ is made ten times smaller."
2. Cell introduction, critical points.
   Before: "Does the gradient vanish at the two hand-derived points, and not
   at a point nearby?"
   After: "The next cell evaluates $\psi$ and the length of its gradient at
   the two critical points and at $(0.3, 0)$." and, after it, "... which only
   repeats the algebra."
3. Cell introduction, numerical test.
   Before: "Does a program classify these points correctly?"
   After: "What does a numerical version of the test report at a flat
   direction? A Hessian assembled from second differences never contains an
   exact zero, so the next cell needs a tolerance below which an eigenvalue
   counts as zero."
4. Cell introduction, cubic error.
   Before: "How fast does the gap grow as we step away from $(-1, 0)$?"
   After: "If the error of the quadratic model is of third order, doubling
   the step should multiply the gap by eight."
5. Voice, edge against total.
   Before: "The two differ because of the skip edge."
   After: "Confusing the two is the usual way to get the multivariable chain
   rule wrong."
6. Voice, ranking.
   Before: "The proposition gives four cases at a critical point:" (bullets)
   After: "Of the three statements, the second is the one to remember, and
   the only one whose proof takes work. The inconclusive case does occur."

## Repetition

Removed or reduced to a clause:

- Cost of probing one weight at a time (four tellings, MVC-1): the
  $1{,}000{,}001$ paragraph of the opening, "the same scaling as probing one
  weight at a time in the opening", and "one backward pass replaces a million
  and one evaluations" are gone. One clause remains at the head of the
  chain-rule part, citing `sec_mdl-matrix-calculus-autodiff`.
- General cost statement "forward mode costs $N$ passes ... backward mode
  costs one" (MVC-15): replaced by references to `sec_mdl-matrix-calculus`
  (count by order, `subsec_mdl-layer-vjp`) and
  `sec_mdl-matrix-calculus-autodiff` (algorithm, measured cost).
- "Partial Derivatives" (MVC-2): limit definition and slice discussion
  reduced to three sentences citing `sec_calculus`; `eq_mdl-part_der` kept.
- Gradient as dot product said three times (MVC-3): once, in the "unique
  vector" form.
- Steepest descent (MVC-5): the four-step recipe, the $\cos\theta$ display
  and "In the $\cos\theta$ form ..." are gone; one clause cites
  `sec_calculus` for the angle argument.
- Clairaut theorem stated twice (MVC-16): once.
- Classification of critical points five times (MVC-19): bullet list
  deleted; the two things it added are in the proposition.
- "We can now ..." recap before the Summary (MVC-22): deleted. The body now
  ends on the last sentence of "Saddles in High Dimension".
- "Tangent Planes": reduced to the equation, the running example and one
  sentence on the normal (label and figure had to stay).
- Second telling of the counterexample $2xy/(x^2+y^2)$ inside the chain rule:
  two sentences with `subsec_mdl-gradient`.

Kept because this file owns them: the sum over paths on the two small graphs;
the first definition of the adjoint; the derivation of the log-sum-exp
Hessian; the counterexample; 24 against 12 multiplications.

One sentence added on purpose (MVC-11): after the hyperbola, the Lagrange
condition for $\mathbf z\cdot\mathbf p - \sum_i p_i\log p_i$ on the simplex,
giving the softmax and the value log-sum-exp, with
:eqref:`eq_mdl-opt-lse-entropy` (the owner is `mdl-lagrangian-duality.md`).
I checked the algebra. The replacement of the hyperbola by this example is
deferred (the figure cannot move in this pass).

## Ranking

- No Proposition block demoted: the five in the file (steepest descent, level
  sets, Lagrange, Clairaut, second-order conditions) are all cited later or
  elsewhere.
- The opening names the three results that carry the section and says that
  tangent planes and Lagrange multipliers can be skipped on a first reading.
- Labels added: `subsec_mdl-chain-rule-paths`, `subsec_mdl-backpropagation`,
  `subsec_mdl-hessian-symmetry`, `subsec_mdl-second-order-taylor` (review,
  structural finding 6).

## Split

Cells would have to move, so the order is unchanged and there are two
markers:

- line 487, before `## The Multivariable Chain Rule`:
  `<!-- split: new section "The Chain Rule on Computational Graphs" starts here -->`
- line 785, before `## The Hessian`:
  `<!-- split: new section "Gradients, Hessians, and Critical Points" resumes here (...) -->`

Why prose alone cannot do it: "Critical Points" holds cell
`multivariable-calculus-a-note-on-mathematical-optimization`, which precedes
the three backpropagation cells, and the Hessian cells follow them.

Reorder needed on the development machine: move the whole block from the
second marker to the end of "Saddles in High Dimension" (cells
`mdl-multivariable-calculus-the-second-order-taylor-approximation-1`,
`multivariable-calculus-hessians`,
`mdl-multivariable-calculus-the-second-derivative-test`) to just before the
first marker. Section one is then: gradient, geometry, critical points,
Lagrange, Hessian, Taylor, test, saddles. Section two is the chain-rule block
(cells `...-the-backpropagation-algorithm-1/2/3`). The moved cells use only
`np`, `lse`, `grad_lse`, `xy0`, `psi` from earlier cells of section one, and
the backpropagation cells use nothing from them, so the move is safe; the new
second file needs its own imports cell. A better order inside section one
(Lagrange last, after saddles, as the review suggests) is a prose-only move
that can be made at the same time.

Done to make the parts stand alone:

- The chain-rule part opens with its own question in three sentences and ends
  on its own bridge to `sec_mdl-matrix-calculus` (the three bridging
  sentences of the old recap, shortened, moved there).
- The Hessian part no longer uses the chain rule (see Correctness), so
  section one does not need section two. Section two cites section one by
  label only (`eq_mdl-nabla_use`, `subsec_mdl-gradient`).
- A sentence at the head of the chain-rule part says that $f$ there denotes
  the output of the graph at hand, not the log-sum-exp (MVC-14).
- The Summary is two paragraphs, each headed by the proposed section title.
- The opening paragraph still announces both halves; its sentence beginning
  "The section then turns to how a gradient is computed" goes to section two
  when the file is cut.

## Cells to cut or redesign on the development machine

- `mdl-multivariable-calculus-directional-derivatives-1` (MVC-4a): four tabs
  confirming $[\tfrac13,\tfrac23]$. Prose reduced to one sentence before and
  one after. Cut, or fold into the first cell as one line.
- `multivariable-calculus-a-note-on-mathematical-optimization` (MVC-4b):
  evaluates a hand-typed gradient at its own roots. The prose now says that
  it only repeats the algebra. Replace by $\psi$ on the circle of radius 2
  against the bound $0.04$, or cut.
- `multivariable-calculus-the-backpropagation-algorithm-3`: autograd
  confirming the hand adjoints. Kept as it was (the review lists the
  three-way agreement and the $2^{24}$ remark among the best); the author may
  still judge it a confirmation cell.

## Other deferred findings

- MVC-11: replace the hyperbola by the entropy-regularized problem and move
  the hyperbola and its figure to the exercises (figure move).
- MVC-14: rename the toy outputs ($f$ of the two graphs). The cells and the
  two figures use `f`; only the scope sentence was added.
- Structural finding 4: delete "Tangent Planes" entirely (needs the label
  `eq_mdl-tangent_plane` and the figure removed).
- Exercises: 5(a) still overlaps the text, which now names two of the three
  functions; exercise 11 (Rayleigh) is still after the extended project;
  exercises 3, 6 and 7 write totals with $\partial$ (correct for $f$ as a
  function of the inputs, which the notation paragraph covers, but exercise 6
  has $\partial x_m/\partial x_0$ on a graph).
- Structural finding 3 (running example dropped in the chain rule): only the
  sentence that the nested squares are a toy with integer values.

## Claims added or changed

- Gradient descent "stops moving exactly where this condition holds" (the
  update is zero if and only if the gradient is).
- The step of gradient descent has length $\eta\|\nabla\ell\|$.
- "Adam, which divides each coordinate of its step by an estimate of that
  coordinate's size, resembles this sign step more than the Euclidean one"
  (MVC-7; a qualitative statement, see `chap_optimization`).
- The Lagrange condition for the entropy problem: $z_i - \log p_i - 1 = \mu$,
  $p_i \propto e^{z_i}$, value log-sum-exp.
- The skip-edge adjoint $\bar p = 7$, $\bar r = 1$.
- "The backward pass ... does the work on the shared part of the graph once."
- "Confusing the two is the usual way to get the multivariable chain rule
  wrong" (a judgment).
- "Of the three statements, the second is the one to remember, and the only
  one whose proof takes work."
- "Second-order information cannot decide this point, and only the tolerance
  makes the program say so" (of $x^2+y^4$ in the numerical test; the finite
  difference returns $2h^2 = 2\times10^{-8}$, below the tolerance $10^{-6}$).
- The chain-rule computation "is the one-variable proof of the chain rule
  with the error measured in norm; `sec_mdl-matrix-calculus` carries it out"
  (checked against the proof of `eq_mdl-jacobian-chain`).

## Slides that now disagree with the body

- "A Million Weights, One Loss, One Step" and its bullet "a million separate
  slopes": the body no longer counts evaluations (the clause "a million
  separate slopes" remains in the opening).
- Steepest-descent slide with $\|\nabla\ell\|\cos\theta$: the body now cites
  `sec_calculus` for the angle argument and proves by Cauchy–Schwarz only.
- Tape/backward slide: "Adjoint $\bar a = \partial f/\partial a = \sum
  \bar c\,\partial c/\partial a$" is the old notation; the body has
  $\bar n$, $\mathrm{d}f/\mathrm{d}n$ in words, and no equation of the
  adjoint with a $\partial$.
- Any slide writing $\partial f/\partial w = 73472$ for the total: the body
  writes $\mathrm{d}f/\mathrm{d}w$.
- Slide kicker "§25.2" is stale numbering (regenerated by `gen_slides.py`).
